import { TestBed } from '@angular/core/testing';

import { GetobservatoriesService } from './getobservatories.service';

describe('GetobservatoriesService', () => {
  let service: GetobservatoriesService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(GetobservatoriesService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
