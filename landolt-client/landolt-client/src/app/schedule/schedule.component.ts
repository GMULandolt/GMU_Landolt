import { Component, OnInit } from '@angular/core';
import { Observatory } from '../observatories/observatory';
import { GetobservatoriesService } from '../observatories/getobservatories.service';

@Component({
  selector: 'app-schedule',
  templateUrl: './schedule.component.html',
  styleUrl: './schedule.component.css'
})
export class ScheduleComponent implements OnInit{

  allObservatories: Observatory[] | undefined;

  constructor(private observatoryService: GetobservatoriesService){};

  ngOnInit(): void {
    this.refreshState();
  }

  refreshState(): void { 
    this.observatoryService.getObservatories()
    .subscribe (obs => {
      this.allObservatories = obs;
      console.log("schedule component")
      console.log(this.allObservatories)
    })
  }
}
